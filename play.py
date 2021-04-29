from keras.preprocessing.sequence import pad_sequences
print("##########################################")
print("#     Start chatting with chatbot :)     #")
print("##########################################")


prepro1 = ""
while prepro1 != 'q':
    
    prepro1 = input("Me : ")
    try:
        prepro1 = clean_text(prepro1)
        prepro = [prepro1]
        
        txt = []
        for x in prepro:
            lst = []
            for y in x.split():
                try:
                    lst.append(vocab[y])
                except:
                    lst.append(vocab['<OUT>'])
            txt.append(lst)
        txt = pad_sequences(txt, 13, padding='post')


        enc_op, stat = enc_model.predict( txt )

        empty_target_seq = np.zeros( ( 1 , 1) )
        empty_target_seq[0, 0] = vocab['<SOS>']
        stop_condition = False
        decoded_translation = ''


        while not stop_condition :

            dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + stat )

            ###########################
            attn_op, attn_state = attn_layer([enc_op, dec_outputs])
            decoder_concat_input = Concatenate(axis=-1)([dec_outputs, attn_op])
            decoder_concat_input = dec_dense(decoder_concat_input)
            ###########################

            sampled_word_index = np.argmax( decoder_concat_input[0, -1, :] )

            sampled_word = inv_vocab[sampled_word_index] + ' '

            if sampled_word != '<EOS> ':
                decoded_translation += sampled_word           


            if sampled_word == '<EOS> ' or len(decoded_translation.split()) > 13:
                stop_condition = True

            empty_target_seq = np.zeros( ( 1 , 1 ) )  
            empty_target_seq[ 0 , 0 ] = sampled_word_index
            stat = [ h , c ] 

        print("chatbot : ", decoded_translation )
        print("=" * 60)

    except:
        print("sorry didn't got you , please type again :( ")